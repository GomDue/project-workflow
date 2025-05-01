class dataset:
    """
    YOLO 모델 학습을 위한 라벨 전처리 클래스.

    주어진 JSON 형식의 어노테이션 파일을 YOLO 포맷(.txt)으로 변환합니다.
    BOX 또는 POLYGON 형태의 어노테이션을 정규화하여 YOLO 입력에 맞게 변환합니다.

    환경 변수로부터 학습/검증 데이터의 입력 폴더 및 출력 폴더 경로를 받아 사용합니다.
    """

    def __init__(self):
        """
        클래스 ID 매핑 딕셔너리 초기화
        """
        CLASS_DICT = [
            "금속캔", "종이", "페트병", "플라스틱", "스티로폼", "비닐", "유리병", "건전지", "형광등"
        ]
        self.class_dict = {c: i for i, c in enumerate(CLASS_DICT)}

    def BoxNormalization(self, image_info, annotation):
        """
        YOLO 포맷을 위한 바운딩 박스 정규화

        Args:
            image_info (dict): 이미지 너비와 높이 정보를 포함
            annotation (dict): BOX 타입 어노테이션

        Returns:
            list: [center_x, center_y, width, height] (YOLO 형식으로 정규화된 값)
        """
        image_height = image_info['IMAGE_HEIGHT']
        image_width = image_info['IMAGE_WIDTH']
        x, y, w, h = annotation['POINTS'][0]

        nx = (x + w / 2) / image_width
        ny = (y + h / 2) / image_height
        nw = w / image_width
        nh = h / image_height

        return [nx, ny, nw, nh]

    def SegementNormalization(self, image_info, annotation):
        """
        YOLO 포맷을 위한 세그먼트 정규화

        Args:
            image_info (dict): 이미지 너비와 높이 정보
            annotation (dict): POLYGON 타입 어노테이션

        Returns:
            list: [x1, y1, x2, y2, ..., xn, yn] (YOLO 형식으로 정규화된 값)
        """
        image_height = image_info['IMAGE_HEIGHT']
        image_width = image_info['IMAGE_WIDTH']
        segments = annotation['POINTS']
        result = []

        for x, y in segments:
            result.append(x / image_width)
            result.append(y / image_height)

        return result

    def Normalization(self, image_info, annotation):
        """
        어노테이션의 타입(BOX or POLYGON)에 따라 정규화 처리

        Args:
            image_info (dict)
            annotation (dict)

        Returns:
            list: YOLO 포맷 정규화 결과
        """
        if annotation['SHAPE_TYPE'] == 'BOX':
            return self.BoxNormalization(image_info, annotation)
        elif annotation['SHAPE_TYPE'] == 'POLYGON':
            return self.SegementNormalization(image_info, annotation)
        else:
            raise ValueError("Unsupported shape type")

    def makeYOLOLabels(self, image_info, annotations):
        """
        여러 어노테이션을 YOLO 라벨 문자열로 변환

        Args:
            image_info (dict)
            annotations (list[dict])

        Returns:
            list[str]: YOLO 라벨 형식 문자열 리스트
        """
        return [
            "{} {}".format(
                self.class_dict[annotation['CLASS']],
                ' '.join(str(p) for p in self.Normalization(image_info, annotation))
            ) for annotation in annotations
        ]

    def preprocess(self) -> None:
        """
        YOLO 학습용 데이터 전처리 실행

        - 환경변수로부터 경로 정보를 불러옴
        - JSON 어노테이션 → YOLO 포맷(.txt)으로 변환하여 저장
        - 학습/검증 데이터 모두 처리
        """
        import os
        import json

        train_dirs = [
            os.environ["YOLO_LABEL__TRAIN_SRC_DIR"],
            os.environ["YOLO_LABEL__TRAIN_DST_DIR"],
        ]
        valid_dirs = [
            os.environ["YOLO_LABEL__VALID_SRC_DIR"],
            os.environ["YOLO_LABEL__VALID_DST_DIR"]
        ]

        for src, dst in [train_dirs, valid_dirs]:
            for file in [f for f in os.listdir(src)]:
                with open(os.path.join(src, file), encoding='UTF8') as f:
                    data = json.load(f)
                    image_info = data['IMAGE_INFO']
                    image_annotations = data['ANNOTATION_INFO']

                    yolo_labels = self.makeYOLOLabels(image_info, image_annotations)

                    yolo_file_name = image_info['FILE_NAME'].split(".")[0]
                    yolo_file_path = os.path.join(dst, yolo_file_name + ".txt")

                    with open(yolo_file_path, 'w') as yolo_file:
                        yolo_file.write('\n'.join(yolo_labels))

            print(f"YOLO 라벨 파일 저장 완료 → {dst}")

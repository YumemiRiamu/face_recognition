``` python
//하르카스케이드 추출
//캠에서 영상추출
let video = document.getElementById('videoInput');
// 비디오 읽기
let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
// 입력행렬
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
// 출력행렬
let gray = new cv.Mat();
// 색변수 생성
let cap = new cv.VideoCapture(video);
// 비디오 불러오기
let faces = new cv.RectVector();
// 하르 카스케이드 내용을 담을 변수
let classifier = new cv.CascadeClassifier();
//카스케이드검출(얼굴검출에 쓰임)
classifier.load('haarcascade_frontalface_default.xml');
//하르 카스케이드 사용
const FPS = 30;
//1초당 30회 프레임
function processVideo() {
    try {
        if (!streaming) {
            // clean and stop.
            src.delete();
            dst.delete();
            gray.delete();
            faces.delete();
            classifier.delete();
            return;
        }
//멈추기
        let begin = Date.now();
//시작
        cap.read(src);
//한프레임씩 영상읽기
        src.copyTo(dst);
//입력영상 복사
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
// 얼굴검출
        classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
// 하르 검출로 검출된 내용을 출력영상에 그리기
        for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);
            cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
        }
// 검출된 얼굴을 직사각형으로 표현
        cv.imshow('canvasOutput', dst);
// output에 이미지 출력
        let delay = 1000/FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    } catch (err) {
        utils.printError(err);
    }
};

// schedule the first one.
setTimeout(processVideo, 0);

```

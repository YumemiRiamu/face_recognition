# 디지털영상처리 보고서

7팀 양용석, 이준성, 이재경

## 딥 러닝을 활용한 얼굴 추출 방법

먼저 딥 러닝을 활용한 얼굴 추출 방법에 대해 알아보기 전에 딥 러닝이란,

머신 러닝의 특정한 한 분야로서 연속된 층에서 점진적으로 의미 있는 표현을 배우는 데 강점이 있으며, 

데이터로부터 표현을 학습하는 새로운 방식입니다.

딥 러닝에서의 딥 이란 단어는 깊은 통찰을 얻을 수 있다는 것을 의미하는 게 아니라 연속된 층으로 표현을 학습한다는 개념을 나타냅니다. 

데이터로부터 모델을 만드는 데 얼마나 많은 층을 사용했는지가 그 모델의 깊이가 됩니다.

최근의 딥 러닝 모델은 표현학습을 위해 수십 개, 수백 개의 연속된 층을 가지고 있습니다.

이 층들을 모두 훈련 데이터에 노출해서 자동으로 학습시킵니다.

딥 러닝에서는 기본 층을 겹겹이 쌓아 올려 구성한 신경망이라는 모델을 사용하여 표현 층을 학습합니다. 

딥 러닝의 일부 핵심 개념이 뇌 구조와 관련은 있지만 뇌를 모델링 한 것은 아닙니다. 

딥러닝은 정보가 연속된 필터를 통과하면서 순도 높게 정제되는 다단계 정보 추출 작업입니다. 

기술적으로는 데이터 표현을 학습하기 위한 다단계 처리 방식을 말합니다. 

이러한 딥러닝을 활용해서 얼굴을 추출하려면 이미지 인식이라는 Convolution Neural Networks, 

즉 CNN이라는 방식으로 이미지를 부분적으로 나눠서 나눠진 이미지의 특징을 추출하고 학습하는 분야입니다.

이미지 인식에서 이미지 라벨링이라는 이미지에 따라 나눠야지 인공지능에 학습을 시킬 수 있기 때문에 다양한 이미지, 데이터를 확보해야 하는데,

이미지넷(IMAGENET)이라는 많은 이미지를 확보한 데이터셋을 활용합니다. 

텐서플로우를 이용한 얼굴인식이 있는데 비교할 얼굴 2장을 Backbone Network에 넣은 뒤,

최종 단에 나오는 Embedding Vector 두 개를 비교하여 같은 사람인지 다른 사람인지 결정하게 됩니다. 

Backbone Network는 보통 CNN 구조이며, Resnet, Inception, VGG 등등 여러 가지 네트워크가 될 수 있습니다.

Embedding Vector는 네트워크 최종 단에 있는 길이가 N인 벡터입니다. 

요약하면, 얼굴 이미지를 1차원 벡터로 정보를 압축하여 두 벡터의 유사도를 비교하는 과정입니다. 

두 벡터의 유사도를 비교하는 방법은 크게 두 가지가 있는데,

두 벡터의 L2 거리를 유사도를 비교하는 triplet-loss라는 것과 두 벡터의 Cosine 거리로 유사도를 비교하는 sofrmax-loss가 있습니다.

이를 활용하여 이미지에서 얼굴을 인식하여 얼굴을 인식하여 나타내는 것입니다.

저희가 사용할 Haar Cascade는 머신 러닝기반의  오브젝트 검출 알고리즘입니다. 

특징(feature)을 기반으로 비디오 또는 이미지에서 오브젝트를 검출하기 위해 사용됩니다. 

직사각형 영역으로 구성되는 특징을 사용기 때문에 픽셀을 직접 사용할 때 보다 동작 속도가 빠릅니다.  

찾으려는 오브젝트(여기에선 얼굴)가  포함된 이미지와 오브젝트가 없는 이미지를 사용하여

Haar Cascade Classifier(하르 특징 분류기)를 학습시킵니다. 

그리고나서 분류기를 사용하여 오브젝트를 검출합니다. 

알고리즘은 다음 4단계로 구성됩니다.

Haar Feature Selection (하르 특징 선택)

Creating  Integral Images (적분 이미지 생성)

Adaboost Training (하르 특징을 사용하여 특징 게산)

Cascading Classifiers (하르 특징을 사용하여 검출)



### code

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


#### 출력 영상
![output](https://user-images.githubusercontent.com/93495684/201446315-362ab9bb-b1bb-49e1-b23f-80d0a3f0f712.gif)

##### 소감

저희 조에서는 opencv에서 제공하는 haar cascades 알고리즘을 사용하여 얼굴 인식을 구현하였습니다. 

위 프로젝트를 통해, haar cascades의 원리와 딥러닝을 활용하는 방법에 대해 알게 되었고, 

캠을 사용해 얼굴 인식까지 시뮬레이션 할 수 있었습니다.

팀원들과 같이 알아가면서 opencv가 더 익숙해지게 되는 시간을 가졌습니다. 

opencv의 함수, 명령어들의 활용도가 매우 높아 활용도 다양한 곳에 할 수 있을 것 같습니다.

몇몇 명령어들은 아직 익숙하지 않지만 시간이 남을때 마다 틈틈이 알아가겠습니다.

다음 팀프로젝트에서도 노력해서 좋은 결과를 만들어내겠습니다. 감사합니다. 



###### 참고자료
1. 딥러닝이란 무엇인가?, 텐서 플로우 블로그 (Tensor ≈ Blog), https://tensorflow.blog/%EC%BC%80%EB%9D%BC%EC%8A%A4-%EB%94%A5%EB%9F%AC%EB%8B%9D/1-%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80/
2. Tensorflow(텐서플로우) - 얼굴인식 Part 1, 라온피플(주), 2019.7.11., https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=laonple&logNo=221583231520#
3. "25분만에 끝내는 인공지능 기초와 활용 및 사례 (ㄹㅇ블루오션)", "유튜브 비디오", 조코딩 JoCoding,  2022. 10. 3., https://www.youtube.com/watch?v=mRnXgBDf_oE 
4. https://docs.opencv.org/5.x/df/d6c/tutorial_js_face_detection_camera.html

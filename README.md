# Airsim을 활용한 강화학습 기반 자율주행
## 프로젝트 개요
Airsim 시뮬레이션 환경을 사용하여 자율주행 차량을 개발하기 위한 강화학습 기반의 소프트웨어입니다. 실제 자율주행 환경을 모방하고, 강화학습 알고리즘을 통해 차량 제어를 학습하는 데 초점을 맞추고 있습니다

## 사용 소프트웨어 및 라이브러리
프로그래밍 언어: Python 3.10<br>
시뮬레이션: Airsim 1.6.0, Coastline 환경<br>
딥러닝 프레임워크: TensorFlow 2.10.0<br>
강화학습 환경 구현: Gym 0.25.2<br>

## 사용법
- 상태변수로 State를 사용할 경우
1. Airsim 시뮬레이션을 실행합니다.<br>
2. ReinforcementLearning\DDPGkerasBasedState\ddpg_main.py을 실행하여 강화학습을 진행합니다. 학습 상황은 TensorBoard를 통해 확인할 수 있습니다.<br>
3. ReinforcementLearning\DDPGkerasBasedState\ddpg_load.py을 실행하여 학습된 모델을 불러와 테스트합니다.<br>
- 상태변수로 State, Image를 사용할 경우
1. Airsim 시뮬레이션을 실행합니다.<br>
2. ReinforcementLearning\DDPGkerasBasedImage\ddpg_main.py을 실행하여 강화학습을 진행합니다. 학습 상황은 TensorBoard를 통해 확인할 수 있습니다.<br>
3. ReinforcementLearning\DDPGkerasBasedImage\ddpg_load.py을 실행하여 학습된 모델을 불러와 테스트합니다.<br>


## 데모
![Demo](docs/demo.gif)
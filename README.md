🍅 Automated Cherry Tomato Harvesting Robot System
Deep Learning 기반 지능형 방울토마토 자동 수확 시스템

본 프로젝트는 농촌 인구 감소 및 고령화 문제 해결을 위해 Computer Vision과 Multi-Tiered Control 기술을 결합하여 개발한 자동화 수확 로봇입니다. 
YOLOv5 기반의 실시간 완숙도 판별과 3차원 좌표 추출을 통해 정밀한 수확 시퀀스를 구현했습니다.

🛠 Tech Stack
Main Control: Raspberry Pi 4 (Python)
Perception: YOLOv5, OpenCV, OpenNI (Orbbec Astra)
Embedded Control: STM32 (Driving), Arduino (Robotic Arm)
Algorithms: Inverse Kinematics (역기구학), PWM Speed Control, Data Filtering Gates

🚀 Key Features
1. Intelligent Perception (Computer Vision)
 -실시간 완숙도 판별: YOLOv5를 활용해 'Red' 토마토를 실시간 탐지하고 중심 좌표를 추출합니다.
Multi-Gate Filtering: 오검출 방지를 위한 4단계 필터링 시스템을 구축했습니다.
 -Orange Tomato : 덜 익은 과실 차단.
 -Skin Gate: 작업자 손(피부색) 인식 및 보호.
 -Size Gate: 실제 지름(18mm ~ sim 60mm) 기반 유효성 검사.
 -Roundness Gate: 원형도 계산을 통한 비정상 형태 객체 제외.

2. Multi-MCU Control System
 -STM32 (주행 제어): PWM 기반의 부드러운 속도 제어 및 조향 보정 기능을 구현했습니다. 탐지 시 즉시 정지 및 5초 무검출 시 재주행 시퀀스를 담당합니다.
 -Arduino (로봇팔 제어): 6축 로봇팔에 역기구학(Inverse Kinematics)을 적용하여, 비전 센서로부터 받은 3차원 좌표로 정밀하게 이동 및 수확합니다.

3. Hardware Optimization
   - 구조 설계: 알루미늄 프레임과 3D 모델링을 활용하여 진동을 최소화한 하중 분산 설계를 적용했습니다.

🔄 System Workflow
1. Autonomous Driving: 로봇이 농로를 주행하며 실시간 영상 분석 수행.
2. Detection & Stop: 완숙 토마토 인식 시 STM32에 stop 신호 전송 및 정지.
3. Coordinate Mapping: 영상 좌표(u, v)와 깊이 정보(Z를 결합하여 3차원 x, y, z 좌표 산출
4. Harvesting: 아두이노 로봇팔이 역기구학 경로를 따라 목표 지점 수확.
5. Resume: 수확 시퀀스 완료 후 다시 주행 재개.

📈 Performance
 인식 정확도 및 수확 성공률: 약 92% 이상 달성 (현장 시연 기준).
 효율성: 수동 수확 대비 일관된 수확 기준 제시 및 노동력 대체 가능성 입증.




https://github.com/user-attachments/assets/b490ea39-e0ca-4855-b04b-a153f40e7df6



# TODO  기능 활용 방법
### 1. 코드에 주석 추가
- 형식: # todo:, # fixme: 
- 파이참은 이 주석을 자동 감지함
### 2.TODO 창 열기:
- `View` > `ToolWindows` > `Todo`
### 3. 필터와 정렬: 
- 프로젝트 범위, 파일별, 키워드별로 TODO를 필터링 및 정렬.
#### 필터링
- Current File: 현재 열려 있는 파일에서만 TODO를 표시합니다.
- Scope: 특정 범위나 디렉토리에 대한 TODO를 표시합니다. 커스텀 스코프를 만들어 특정 파일들에 대한 TODO만 볼 수 있습니다.
- Type: TODO와 FIXME를 분리하여 볼 수 있습니다.
#### 정렬
TODO 항목을 파일 경로나 항목 내용에 따라 정렬할 수 있습니다. 예를 들어, 파일 경로로 정렬하면 프로젝트 구조에 따라 TODO가 정리됩니다.
#### 클릭하여 바로 이동:
TODO 항목을 클릭하면 해당 주석이 있는 코드로 바로 이동합니다. 이를 통해 해당 부분의 코드를 바로 확인하고 수정할 수 있습니다.
### 5. Editor > TODO에서 사용자 정의 패턴 설정 가능.:
PyCharm에서는 TODO 외에도 다른 커스텀 패턴을 인식하도록 설정할 수 있습니다. 예를 들어, # NOTE:나 # BUG: 같은 다른 키워드도 TODO 리스트에 포함시킬 수 있습니다.

### 6. 커스텀 패턴 설정 방법:
1. 상단 메뉴에서 File > Settings(Windows/Linux) 또는 PyCharm > Preferences(macOS)로 이동합니다.
2. Editor > TODO 섹션으로 이동합니다.
3. 기본적으로 TODO와 FIXME 패턴이 설정되어 있으며, 여기에 새로운 패턴을 추가할 수 있습니다.
- 예를 들어, # NOTE:나 # BUG:와 같은 패턴을 추가하고 싶다면 `Regex` 필드에 해당 키워드를 입력합니다.
4. 설정을 저장하면, 해당 패턴을 가진 주석도 TODO 창에서 관리할 수 있게 됩니다.

### 7. 자동 완성: 
TODO 주석 작성 시 PyCharm이 자동 완성 지원.


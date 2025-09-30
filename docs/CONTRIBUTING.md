# 기여 가이드

QuAP-FL 프로젝트에 기여해주셔서 감사합니다. 이 문서는 프로젝트에 기여하는 방법을 안내한다.

## 목차

- [시작하기](#시작하기)
- [개발 환경 설정](#개발-환경-설정)
- [기여 프로세스](#기여-프로세스)
- [코딩 규칙](#코딩-규칙)
- [테스트](#테스트)
- [커밋 메시지](#커밋-메시지)
- [이슈 리포팅](#이슈-리포팅)

## 시작하기

### 기여 유형

다음과 같은 기여를 환영한다:

- **버그 리포트**: 버그를 발견하면 이슈로 등록
- **기능 제안**: 새로운 기능 아이디어 제안
- **코드 기여**: 버그 수정, 기능 구현, 성능 개선
- **문서 개선**: 오타 수정, 예제 추가, 설명 개선
- **테스트**: 테스트 커버리지 향상

### 행동 강령

본 프로젝트는 모든 기여자가 존중받는 환경을 유지한다. 다음을 준수해야 한다:

- 건설적이고 존중하는 피드백
- 다양한 관점과 경험 존중
- 프로젝트와 커뮤니티의 이익 우선

## 개발 환경 설정

### 1. 저장소 포크 및 클론

```bash
# 포크 후 클론
git clone https://github.com/your-username/quap-fl.git
cd quap-fl

# 업스트림 추가
git remote add upstream https://github.com/original-owner/quap-fl.git
```

### 2. 가상환경 생성

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 개발 의존성 설치

```bash
pip install pytest pytest-cov black flake8 mypy
```

### 4. 구현 검증

```bash
python main.py --validate_only
```

## 기여 프로세스

### 1. 이슈 생성

먼저 작업하려는 내용에 대한 이슈를 생성하거나 기존 이슈를 찾는다.

```
예시 이슈 제목:
- [Bug] 클리핑 값이 음수가 될 수 있음
- [Feature] FEMNIST 데이터셋 지원 추가
- [Docs] API 문서 예제 추가
```

### 2. 브랜치 생성

```bash
git checkout -b feature/your-feature-name
# 또는
git checkout -b fix/your-bug-fix
```

브랜치 명명 규칙:
- `feature/`: 새로운 기능
- `fix/`: 버그 수정
- `docs/`: 문서 개선
- `test/`: 테스트 추가
- `refactor/`: 리팩터링

### 3. 코드 작성

코딩 규칙을 준수하며 코드를 작성한다 (아래 참조).

### 4. 테스트 작성

새로운 기능에는 반드시 테스트를 포함해야 한다.

```bash
# 단위 테스트 실행
python -m pytest tests/

# 특정 테스트
python tests/test_tracker.py

# 커버리지
pytest --cov=framework tests/
```

### 5. 코드 검증

```bash
# 포매팅 체크
black --check .

# 린트
flake8 framework/ models/ data/

# 타입 체크
mypy framework/
```

### 6. 커밋

```bash
git add .
git commit -m "feat: add FEMNIST dataset support"
```

### 7. 푸시 및 PR 생성

```bash
git push origin feature/your-feature-name
```

GitHub에서 Pull Request를 생성한다.

## 코딩 규칙

### Python 스타일

- **PEP 8** 준수
- **Black** 포매터 사용 (line length: 100)
- **Type hints** 사용 권장

```python
def compute_privacy_budget(self, participation_rate: float) -> float:
    """
    참여율에 따른 프라이버시 예산 계산

    Args:
        participation_rate: 클라이언트 참여율 (0.0 ~ 1.0)

    Returns:
        적응형 프라이버시 예산
    """
    pass
```

### 문서화

- **Docstring**: Google 스타일 사용
- **주석**: 복잡한 로직에 설명 추가
- **예제**: 가능한 경우 사용 예제 포함

### 변수 명명

- **클래스**: PascalCase (`ParticipationTracker`)
- **함수/메소드**: snake_case (`compute_privacy_budget`)
- **상수**: UPPER_SNAKE_CASE (`EPSILON_BASE`)
- **Private**: 밑줄 접두사 (`_internal_method`)

### 파일 구조

```python
# 파일 상단에 경로 명시
# framework/participation_tracker.py

"""
모듈 설명 (1-3줄)
"""

# Imports
import standard_library
import third_party
from local import module

# Constants
CONSTANT_VALUE = 1.0

# Classes and Functions
class MyClass:
    pass
```

## 테스트

### 테스트 작성 가이드

1. **단위 테스트**: 각 함수/메소드를 독립적으로 테스트
2. **통합 테스트**: 여러 컴포넌트 상호작용 테스트
3. **검증 테스트**: 전체 시스템 동작 검증

### 테스트 예제

```python
import unittest
from framework.participation_tracker import ParticipationTracker

class TestParticipationTracker(unittest.TestCase):
    def test_initialization(self):
        """초기화 테스트"""
        tracker = ParticipationTracker(10)
        self.assertEqual(tracker.num_clients, 10)

    def test_update(self):
        """참여 기록 업데이트 테스트"""
        tracker = ParticipationTracker(10)
        tracker.update([0, 1, 2])
        self.assertEqual(tracker.total_rounds, 1)
```

### 커버리지 목표

- **전체 커버리지**: 80% 이상
- **핵심 모듈**: 90% 이상
- **새로운 기능**: 100%

## 커밋 메시지

[Conventional Commits](https://www.conventionalcommits.org/) 규칙을 따른다.

### 형식

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포매팅
- `refactor`: 리팩터링
- `test`: 테스트 추가/수정
- `chore`: 빌드, 설정 변경

### 예제

```
feat(privacy): add Rényi DP support

- Implement Rényi divergence calculation
- Add privacy accountant for Rényi DP
- Update tests

Closes #42
```

```
fix(clipping): prevent negative clip values

Clip values could become negative when all gradients
are zero. Added minimum clip value constraint.

Fixes #38
```

## 이슈 리포팅

### 버그 리포트 템플릿

```markdown
## 버그 설명
명확하고 간결한 버그 설명

## 재현 방법
1. '...'로 이동
2. '...'를 클릭
3. '...'까지 스크롤
4. 오류 발생

## 예상 동작
무엇이 일어나야 하는지 설명

## 실제 동작
실제로 무엇이 일어났는지 설명

## 환경
- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.8.10]
- PyTorch: [e.g. 2.0.0]
- QuAP-FL: [e.g. 1.0.0]

## 추가 컨텍스트
스크린샷, 로그 등
```

### 기능 제안 템플릿

```markdown
## 기능 설명
명확하고 간결한 기능 설명

## 동기
이 기능이 왜 필요한가?

## 제안 방법
기능이 어떻게 동작해야 하는가?

## 대안
고려한 다른 방법들

## 추가 컨텍스트
관련 논문, 코드 예제 등
```

## 질문

기여와 관련된 질문이 있으면:

- GitHub Issues에 질문 이슈 생성
- 이메일: your.email@example.com

## 감사합니다

모든 기여자에게 감사드립니다. 여러분의 시간과 노력이 QuAP-FL을 더 좋게 만듭니다!
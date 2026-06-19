# ANI Server

FastAPI 기반 문서 분석 서버입니다.

텍스트 입력 또는 파일 업로드로 받은 문서를 Azure AI Search 하이브리드 검색과 Azure OpenAI 분석 Agent를 사용해 문서 유형, 문서 성격, 포함 정보, 핵심 수치, 법적/주의 문구로 정리합니다.

## 주요 기능

- 문서 정보 분석: `POST /documents/analyze/text`, `POST /documents/analyze/file`
- 구조화 번역: `POST /documents/translate`
- 지원 파일 추출: `txt`, `pdf`, `docx`, `png`, `jpg`, `jpeg`

## 문서 분석 API

Azure AI Search 문서 인덱스에서 키워드 검색과 벡터 검색을 함께 수행한 뒤, Azure OpenAI가 문서 유형/성격/포함 정보/핵심 수치/법적 및 주의 문구를 분석합니다.

### 텍스트 분석

```http
POST /documents/analyze/text
Content-Type: application/json
```

```json
{
  "text": "외국인 고객 대상 예금 상품 안내. 기본금리, 가입기간, 우대조건, 예금자보호 문구 포함",
  "top": 5
}
```

### 파일 분석

```http
POST /documents/analyze/file
Content-Type: multipart/form-data
```

필드:

- `file`: 분석할 파일
- `top`: 검색 결과 개수, 기본값 5

이미지 파일(`png`, `jpg`, `jpeg`)은 Azure Vision OCR로 텍스트를 추출한 뒤 분석하고, 그 외 `txt`, `pdf`, `docx`는 기존 파일 텍스트 추출 로직을 사용합니다.

응답에는 `analysis`와 AI Search에서 가져온 `retrieved_documents`가 함께 포함됩니다. `analysis.key_numbers_preview`에는 원문에서 확인된 금리, 가입기간, 예금자보호 한도, 중도해지 이율 같은 핵심 수치가 포함됩니다.

```json
{
  "analysis": {
    "key_numbers_preview": [
      {
        "label": "기본금리",
        "value": "연 3.2%",
        "source_text": "기본금리 연 3.2%"
      }
    ]
  }
}
```

## 구조화 번역 API

문서 분석 결과와 사용자가 선택한 대상 언어/톤/금융 용어를 바탕으로 언어별 번역 결과를 생성합니다.

번역 결과는 4페이지 검토 화면에서 바로 렌더링할 수 있도록 제목, 요약, 섹션 단위로 반환됩니다.

```http
POST /documents/translate
Content-Type: application/json
```

```json
{
  "source_text": "기본금리 연 3.20%, 가입기간 12개월",
  "document_analysis": {
    "document_type": "금융 상품안내문",
    "key_numbers_preview": [
      {
        "label": "기본금리",
        "value": "연 3.20%",
        "source_text": "기본금리 연 3.20%"
      }
    ]
  },
  "target_languages": ["영어 (English)", "베트남어 (Tiếng Việt)"],
  "tone_style": "공식적이고 신뢰감 있는 금융 문체",
  "finance_terms": []
}
```

응답:

```json
{
  "translations": [
    {
      "language": "영어 (English)",
      "language_code": "en",
      "title": "",
      "summary": "",
      "sections": [
        {
          "id": "section_1",
          "order": 1,
          "source_label": "기본금리",
          "source_text": "기본금리 연 3.20%",
          "translated_label": "Base Interest Rate",
          "translated_text": "3.20% p.a."
        }
      ],
      "full_text": ""
    }
  ]
}
```

## 초기 세팅

1. 가상환경을 생성하고 활성화합니다.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

CMD:

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

2. 필요한 라이브러리를 설치합니다.

PowerShell/CMD:

```powershell
pip install -r requirements.txt
```

3. 환경변수 파일을 준비합니다.

PowerShell:

```powershell
Copy-Item .env.example .env
```

CMD:

```bat
copy .env.example .env
```

4. `.env`에 Azure AI Search, Azure OpenAI, Azure AI Vision 값을 입력합니다.

5. 서버를 실행합니다.

PowerShell/CMD:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 참고

- 기본 CORS 허용 origin은 `http://localhost:5173`입니다.

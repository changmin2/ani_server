# ANI Server

FastAPI 기반 번역/OCR 서버입니다.

텍스트, 파일, 이미지에서 추출한 문장을 Azure Translator로 1차 번역하고, Azure AI Search 용어집과 Azure OpenAI를 사용해 금융/은행 문맥에 맞게 번역 품질을 보정합니다. 이미지 OCR은 Azure AI Vision을 사용하며, 번역된 문구를 이미지 위에 다시 그린 결과도 반환합니다.

## 주요 기능

- 텍스트 번역: `POST /translate/text`
- 파일 번역: `POST /translate/file`
- 이미지 OCR 및 번역: `POST /ocr/image`
- 지원 파일 추출: `txt`, `pdf`, `docx`, `png`, `jpg`, `jpeg`

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

4. `.env`에 Azure Translator, Azure AI Search, Azure OpenAI, Azure AI Vision 값을 입력합니다.

5. 서버를 실행합니다.

PowerShell/CMD:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 참고

- `pytesseract`를 사용하는 파일 OCR 기능은 별도의 Tesseract OCR 설치가 필요할 수 있습니다.
- 기본 CORS 허용 origin은 `http://localhost:5173`입니다.

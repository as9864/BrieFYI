"""이메일 발송 도구 (구현 항목 #6). Resend API 사용 예시.
SMTP를 쓰려면 send_email 내부만 smtplib 호출로 교체하면 되고, 그 외 노드는 그대로 둔다.
"""
import requests

from config import config

RESEND_URL = "https://api.resend.com/emails"


def send_email(subject: str, html: str) -> dict:
    if not config.RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY가 설정되지 않았습니다 (.env 확인)")
    if not config.EMAIL_FROM or not config.EMAIL_TO:
        raise RuntimeError("EMAIL_FROM / EMAIL_TO가 설정되지 않았습니다 (.env 확인)")

    resp = requests.post(
        RESEND_URL,
        headers={"Authorization": f"Bearer {config.RESEND_API_KEY}"},
        json={
            "from": config.EMAIL_FROM,
            "to": [config.EMAIL_TO],
            "subject": subject,
            "html": html,
        },
        timeout=15,
    )
    if not resp.ok:
        # Resend가 본문에 실패 사유(예: 도메인 미인증, 수신자 제한)를 담아주므로 그대로 노출한다.
        raise RuntimeError(f"Resend 발송 실패 ({resp.status_code}): {resp.text}")
    return resp.json()

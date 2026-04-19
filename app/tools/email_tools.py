import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)


async def send_email(to: str, subject: str, body: str) -> dict:
    """
    Sends an email via Gmail SMTP.
    Use when user asks to send an email or follow up with someone.
    Requires GMAIL_USER and GMAIL_APP_PASSWORD in .env.

    Parameters:
        to (str): Recipient email address
        subject (str): Email subject line
        body (str): Email body text

    Returns:
        dict: success (bool) and message (str)
    """
    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_password:
        return {"success": False, "message": "Gmail credentials not configured in .env"}
    try:
        msg = MIMEMultipart()
        msg["From"] = gmail_user
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=465,
            use_tls=True,
            username=gmail_user,
            password=gmail_password,
        )

        logger.info(f"Email sent to {to}: {subject}")
        return {"success": True, "message": f"Email sent successfully to {to}"}

    except Exception as e:
        logger.error(f"send_email error: {e}")
        return {"success": False, "message": f"Failed to send email: {str(e)}"}

import mimetypes
import ssl
from email.message import EmailMessage
import smtplib
from Keypass.key_pass import KEYPASS

def setAutoemail(emails: list, subject, body_html, image_buffers=None, image_cids=None, attachment=False):
    import mimetypes
    email_sender = emails[0]
    email_receiver = emails[1]
    email_password = KEYPASS[email_sender]

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content("This is the plain text version of the email.")
    em.add_alternative(body_html, subtype='html')

    if image_buffers and image_cids:
        for image_buffer, image_cid in zip(image_buffers, image_cids):
            em.get_payload()[1].add_related(
                image_buffer.read(),
                maintype='image',
                subtype='png',
                cid=f"<{image_cid}>"
            )

    if attachment:
        file_path = 'report.pdf'
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type, mime_subtype = mime_type.split('/')

        with open(file_path, 'rb') as f:
            em.add_attachment(f.read(),
                              maintype=mime_type,
                              subtype=mime_subtype,
                              filename='report.pdf')

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.send_message(em)

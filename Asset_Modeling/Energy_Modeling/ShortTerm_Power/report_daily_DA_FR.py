import smtplib
from email.message import EmailMessage

# Email details
sender = 'hugo.lambert.perso@gmail.com'
password = 'Sasha200'  # Use an "App Password" if 2FA is enabled
recipient = 'hugo.lambert.perso@gmail.com'
subject = 'Test Email from Python'
body = 'Hello, this is a test email sent from Python!'

# Create the email
msg = EmailMessage()
msg['From'] = sender
msg['To'] = recipient
msg['Subject'] = subject
msg.set_content(body)

# Send the email
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
    smtp.login(sender, password)
    smtp.send_message(msg)

print("Email sent successfully!")
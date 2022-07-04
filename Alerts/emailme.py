import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def email_alert(live_feed_link):
    mail_content = "we have detected fire at your premisis. You can check the live feed of the situation here:\n" + \
        live_feed_link + "  snapshot is attached from live camera for your reference"
    # The mail addresses and password
    sender_address = 'Your sending mail address'
    sender_pass = 'your sender's password'
    receiver_address = 'your receiver's email'
    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Fire Alert!'
    fire_snap_filename = "fire.jpg"
    attachment = open(fire_snap_filename, 'rb')
    obj = MIMEBase('application', 'octet-stream')
    obj.set_payload((attachment).read())
    encoders.encode_base64(obj)
    obj.add_header('Content-Disposition',
                   "attachment; filename= "+fire_snap_filename)

    # The subject line
    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    message.attach(obj)
    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    # login with mail_id and password
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')

"""
functions around internets
"""

from __future__ import absolute_import

try:
    from .secret.secret import *
except:
    from .secret.dummy import *

import os
import urllib.request
import urllib.parse
try:
    import requests
except:
    requests = None
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

# Exceptions
class InternetException(Exception):
    """Base exeption for internet"""
def _raise_no_notify_token_exception():
    raise InternetException("No token for Notify")
def _raise_no_email_acount_exception():
    raise InternetException("Not enough email acount information")
def _raise_no_module_exception(module, recommend=None):
    if recommend:
        message = "You do not have \'{}\' installed. \nYou can use \'{}\' instead or install the module.".format(module, recommend)
    else:
        message = "You do not have \'{}\' installed.".format(module)
    raise InternetException(message)

def send_line(
    message:str,
    file_path:dict=None,
    token=LINE_NOTIFY_TOKEN
):
    """
    Send line, using requests.
    TODO: support line stamps

    argument
        message : str
            The message you want to send.
        file_path : str (default : None)
            The path to a file you want to send.
        token : str (default : LINE_NOTIFY_TOKEN)
            The token that you have registered
    """
    if not token:
        _raise_no_notify_token_exception()
    if not requests:
        _raise_no_module_exception(module="requests", recommend="send_line_message()")
    
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization" : "Bearer " + token}
    payload = {"message" : message}

    if file_path:
        abs_path = os.path.abspath(file_path)
        files = {"imageFile" : open(abs_path, "rb")}
        requests.post(url, headers=headers, params=payload, files=files)
    else:
        requests.post(url, headers=headers, params=payload)


def send_line_message(
    message:str,
    token=LINE_NOTIFY_TOKEN
):
    """
    Send line message.
    Does not support sending images.

    arguments:
        message : str
            The message you want to send, using notify
        token : str (deafult : LINE_NOTIFY_TOKEN)
            The token that you have registered
    """
    if not token:
        _raise_no_notify_token_exception()
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization" : "Bearer " + token}
    method = "POST"

    payload = {"message" : message}

    try:
        payload = urllib.parse.urlencode(payload).encode("utf-8")
        req = urllib.request.Request(url=url, data=payload, method=method, headers=headers)
        urllib.request.urlopen(req)
    except Exception as e:
        print("Exception on Notify : ", e)

def send_email(
    message,
    subject,
    to_addr,
    from_addr=ADDRESS,
    password=PASSWORD
):
    """
    Send email.
    Does not support file attachment.
    Does not support Bcc address.
    (Because I don't use them...)

    argument
        message : str
            The message you want to send
        subject : str
            The Subject of the email
        to_addr : str
            The email-address you want to send the message to
        from_addr : str (default : ADRESS)
            Your address
        password : str (default : PASSWORD)
            Your email-acount password
    """
    if not from_addr or not password:
        _raise_no_email_acount_exception()

    msg = MIMEText(message)
    msg["To"]      = to_addr
    msg["From"]    = from_addr
    msg["Subject"] = subject
    msg["Date"]    = formatdate()

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(ADDRESS, PASSWORD)
    server.sendmail(from_addr, to_addr, msg.as_string())
    server.close()
    

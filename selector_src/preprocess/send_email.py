import datetime as dt
import json
import sys
import smtplib
import time
from email.message import EmailMessage


def send_mail(to_mail, title, content):
    """
    发送邮件
    Args:
        to_mail: 接收邮件方
        title: 标题
        content: 正文信息
    Returns:
        发送是否成功, True/False
    """
    try:
        msg = EmailMessage()
        msg['Subject'] = title
        msg['From'] = '2580419315@qq.com'
        msg['To'] = 'qingfengwuyu233@163.com'
        msg.set_content(content)
        server = smtplib.SMTP_SSL('smtp.qq.com')
        server.login('2580419315@qq.com', "ysvsqltpkqcsdjca")
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == '__main__':
    title = sys.argv[1]
    content = sys.argv[2]
    send_mail('qingfengwuyu233@163.com', title=title, content=content)


import winsound
import playsound
from Alerts.make_call import requestCall


def raise_alarm():
    playsound.playsound(r'C:\Users\z004fznd\Desktop\Major\Alerts\FireAlarm.mp3',
                        winsound.SND_LOOP + winsound.SND_ASYNC)
    requestCall()

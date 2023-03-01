from flask import Flask
from flask import render_template
from flask_sock import Sock
from flask import Response
import cv2



app = Flask(__name__)
sock = Sock(app)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# reciclap = cv2.CascadeClassifier(cv2.data.haarcascades +
#     "haarcascade_eye.xml")


reciclap = cv2.CascadeClassifier('botella.xml')


def generate():
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #app = reciclap.detectMultiScale(gray, 1.3, 5)
            app = reciclap.detectMultiScale(gray,
                                            scaleFactor=5,
                                            minNeighbors=91,
                                            minSize=(70, 78))
            for (x, y, w, h) in app:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Plastico', (x, y-10),
                            2, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            #cv2.imshow('frame', frame)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                  bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")

@sock.route('/echo')
def echo(sock):
    while True:
        data = sock.receive()
        sock.send(data)


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    cache.cached()


@app.route('/metasploit', methods=['GET', 'POST'])  # ['GET', 'POST']
def metasploit():
    titulo = "Reciclapp"
    return render_template('metasploit/index.html',
                           titulo=titulo,)


@app.route('/netcat', methods=['GET', 'POST'])  # ['GET', 'POST']
def netcat():
    titulo = "Reciclapp"
    return render_template('netcat/index.html',
                           titulo=titulo,)


# if __name__ == "__main__":
#    app.run(debug=False)
if __name__ == '__main__':
    app.run(host='0.0.0.0')
cap.release()
# cap.release()

from flask import Flask,render_template,request,redirect,url_for,session,flash
import tensorflow as tf

app=Flask(__name__,static_folder='static',template_folder='template')
model=tf.keras.models.load_model("model/TrashNetModel.keras")
app.secret_key="my key"

@app.route('/')
def home():
    if session.get('status') == 'failure':
        flash('image should have a higher resolution')
    if session.get('status') != 'normal':
        session['status']='normal'
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.files)
    image=request.files["image"]
    image=image.read()
    image=tf.image.decode_image(image)
    image=tf.expand_dims(image,axis=0)

    if image.shape[1]<224 or image.shape[2]<224:
        print(image.shape)
        session['status']='failure'
        return redirect(url_for('home'))
    
    image=tf.image.resize(image,(224,224))

    pred=model.predict(image)
    pred=tf.argmax(pred[0]).numpy()
    return render_template('predict.html',pred=pred)

if __name__=="__main__":
    app.run()
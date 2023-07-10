from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded file
        image_file = request.files['image']
        # Save the file to a location
        image_path = './uploads/' + image_file.filename
        image_file.save(image_path)
        # Render the template with the uploaded image path
        return render_template('upload.html', image_path=image_path)
    # Render the initial upload form
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

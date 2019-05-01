import face_recognition
from PIL import Image, ImageDraw

# Image of known person and encoding them

# Person 1
image_of_bill = face_recognition.load_image_file('./img/known/Bill Gates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

# Person 2
image_of_steve = face_recognition.load_image_file('./img/known/Steve Jobs.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

# Preson 3
image_of_himanshu = face_recognition.load_image_file('./img/known/Hima.jpeg')
himanshu_face_encoding = face_recognition.face_encodings(image_of_himanshu)[0]

# Person 4 
image_of_nitin = face_recognition.load_image_file('./img/known/nitin.jpeg')
nitin_face_encoding = face_recognition.face_encodings(image_of_nitin)[0]

# Person 5 
image_of_sumit = face_recognition.load_image_file('./img/known/sumit.jpeg')
sumit_face_encoding = face_recognition.face_encodings(image_of_sumit)[0]

# Person 6 
image_of_chetan = face_recognition.load_image_file('./img/known/chetan.jpeg')
chetan_face_encoding = face_recognition.face_encodings(image_of_chetan)[0]

# Person 7 
image_of_jeewan = face_recognition.load_image_file('./img/known/G1.jpeg')
jeewan_face_encoding = face_recognition.face_encodings(image_of_jeewan)[0]

# # Person 8 
image_of_sidhant = face_recognition.load_image_file('./img/known/sidhant.jpeg')
sidhant_face_encoding = face_recognition.face_encodings(image_of_sidhant)[0]

# # Person 9
image_of_dushyant = face_recognition.load_image_file('./img/known/dushyant.jpeg')
dushyant_face_encoding = face_recognition.face_encodings(image_of_dushyant)[0]


# Create array of encodings and names of known person

known_face_encoding = [
    bill_face_encoding,
    steve_face_encoding, 
    sumit_face_encoding, 
    jeewan_face_encoding,
    sidhant_face_encoding,
    himanshu_face_encoding, 
    nitin_face_encoding,
    dushyant_face_encoding,
    chetan_face_encoding
]

known_face_names = [
    "  Bill Gates",
    "  Steve Jobs",
    " Sumit ",
    " G1 ",
    " Siddhant Sir", 
    " Himanshu Namdev ",
    " Nitin ",
    " Dushyant Sir ",
    " Chetan C "
]

# Load test image to find faces in 

test_image = face_recognition.load_image_file('./img/groups/panwar_group_2.jpeg')
# test_image = face_recognition.load_image_file('./img/groups/g1.jpeg')


# Find faces in test image

face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format

pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance of pil_image

draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image

for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    # Give result in True and False
    matches = face_recognition.compare_faces(known_face_encoding, face_encoding, tolerance = 0.46)
    
    name = "Unknown Preson"

    # if Match 
    if True in matches:
        first_match_index = matches.index(True)

        name = known_face_names[first_match_index]
    
    # Draw the box 

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))

    # Draw the label

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom-text_height - 10), (right, bottom)), fill = (255, 0, 0), outline = (0, 0, 0))
    draw.text((left+6, bottom-text_height-5), name, fill = (255, 255, 255, 255))

# Delete/Free draw object 
del draw

# Display Image
pil_image.show()

# Save Image
pil_image.save('identify.jpeg')


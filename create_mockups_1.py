
######### SOLUTION 1 DRAFT but useful ############

from PIL import Image, ImageFont

template = Image.open('./images/templates/template.jpg')
image = Image.open('./images/image.jpg')

width, height = template.size
image = image.resize((783, 982))

mockup = Image.new('RGB', template.size, (255, 255, 255))

mockup.paste(template, (0, 0))

x = 400
y = 300

mockup.paste(image, (x, y))

mockup.save('./images/outputs/mockup_from_solution1.png')
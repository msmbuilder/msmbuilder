# Author: Muneeb Sultan <msultan@stanford.edu>
# Contributors: Matthew Harrigan <matthew.harrigan@outlook.com>
# Copyright (c) 2016, Stanford University
# All rights reserved.


def show_markov_appreciation():
    from PIL import Image
    import requests
    from io import BytesIO
    response = requests.get("https://upload.wikimedia.org/wikipedia/commons/"
                            "thumb/7/70/AAMarkov.jpg/330px-AAMarkov.jpg")
    img = Image.open(BytesIO(response.content))
    img.show()

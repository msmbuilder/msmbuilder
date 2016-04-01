# Author: Muneeb Sultan <msultan@stanford.edu>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from PIL import Image
import requests
from io import BytesIO
def show_markov_appreciation():
    response = requests.get("https://upload.wikimedia.org/wikipedia/commons/"
                            "thumb/7/70/AAMarkov.jpg/330px-AAMarkov.jpg")
    img = Image.open(BytesIO(response.content))
    img.show()

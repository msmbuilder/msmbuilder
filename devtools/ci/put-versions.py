from __future__ import print_function
import os
import boto
from boto.s3.key import Key
import msmbuilder.version
import sys

if len(sys.argv) != 2:
    print("give me a file")
    sys.exit()

# The secret key is available as a secure environment variable
# on travis-ci to push the build documentation to Amazon S3.
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
BUCKET_NAME = 'msmbuilder.org'

bucket_name = AWS_ACCESS_KEY_ID.lower() + '-' + BUCKET_NAME
conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY)
bucket = conn.get_bucket(BUCKET_NAME)

k = Key(bucket)
k.key = 'versions.json'
k.set_contents_from_filename(sys.argv[1])

import os
import boto
from boto.s3.key import Key
import msmbuilder.version

# The secret key is available as a secure environment variable
# on travis-ci to push the build documentation to Amazon S3.
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
BUCKET_NAME = 'msmbuilder.org'

bucket_name = AWS_ACCESS_KEY_ID.lower() + '-' + BUCKET_NAME
conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY)
bucket = conn.get_bucket(BUCKET_NAME)

root = 'doc/_build'

if msmbuilder.version.release:
    versions = json.load(urllib2.urlopen('http://www.msmbuilder.org/versions.json'))

    # new release so all the others are now old
    for i in xrange(len(versions)):
        versions[i]['latest'] = False

    versions.append({'version' : msmbuilder.version.short_version, 'latest' : True})

    k = Key(bucket)
    k.key = 'versions.json'
    k.set_contents_from_string(json.dumps(versions))
    


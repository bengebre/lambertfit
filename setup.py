import setuptools

setuptools.setup(
    name='lambertfit',
    version='0.1',
    author='Ben Engebreth',
    author_email='ben.engebreth@gmail.com',
    description='',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/bengebre/lambertfit',
    project_urls = {'Issues': 'https://github.com/bengebre/lambertfit/issues'},
    license='MIT',
    packages=['lambertfit'],
    install_requires=['poliastro'],
)

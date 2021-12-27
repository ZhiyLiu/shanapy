from distutils.core import setup
setup(
  name = 'shanapy',
  packages = ['shanapy'],
  version = '0.1', 
  license='MIT',   
  description = 'Anatomical shape analysis with s-reps',   # Give a short description about your library
  author = 'Zhiyuan Liu',          
  author_email = 'zhiy@cs.unc.edu',
  url = 'https://github.com/ZhiyLiu/',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ZhiyLiu/shanapy/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Shape analysis', 'Skeletal representation'],   # Keywords that define your package best
  install_requires=[            
          'vtk',
          'numpy',
          'pyvista'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Shape analysis :: Simulation',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
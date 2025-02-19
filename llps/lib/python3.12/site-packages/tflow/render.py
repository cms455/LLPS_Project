import os
from mako.template import Template

def render(name, version):
    template = Template(
        filename='{dir_path}/gitlab-ci.tmpl'.format(
            dir_path=os.path.dirname(os.path.realpath(__file__))
        ),
    )
    
    with open('.gitlab-ci.yml', 'w') as f:
        f.write(
            template.render(mlflow_model_name=name, mlflow_model_version=version))

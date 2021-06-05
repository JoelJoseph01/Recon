from flask import render_template
from flask_login import UserMixin

from app import db, login_manager
from app.refer import blueprint

@blueprint.route('/<template>')
def send_template(template):
    return render_template(template)
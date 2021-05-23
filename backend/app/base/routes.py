from flask import render_template
from flask_login import UserMixin

from app import db, login_manager
from app.base import blueprint


@blueprint.route('/')
def route_default():
    return render_template('index.html')

@blueprint.route('/<template>')
def send_template(template):
    return render_template(template)

@login_manager.user_loader
def load_user(id):
    return id

# Error handlers
@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('page-403.html'), 403

@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html'), 403

@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html'), 404

@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('page-500.html'), 500
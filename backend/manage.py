from flask_migrate import Migrate

from config import config_dict
from app import create_app, db


# The configuration
DEBUG = config('DEBUG', default=True, cast=bool)
get_config_mode = 'Debug' if DEBUG else 'Production'
try:
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')


app = create_app( app_config )
migrate = Migrate(app, db)

# manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
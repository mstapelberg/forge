import pytest
import yaml
import tempfile
from pathlib import Path
from forge.core.database import DatabaseManager

@pytest.fixture(scope="module")
def db_config():
    """Create test database configuration"""
    # AWS database credentials
    db_config = {
        'database': {
            'dbname': 'test_database',  # The name of the database you created on RDS
            'user': 'myless',           # Replace with your RDS username
            'password': 'vcrtiwzr',     # Replace with your RDS password
            'host': 'database-vcrtiwzr.cfg4i4qmuc4m.us-east-1.rds.amazonaws.com',
            'port': 5432                # Default Postgres port unless you changed it
        }
    }
    return db_config

@pytest.fixture(scope="module")
def db_manager(db_config):
    """Create database manager connected to AWS database"""
    with tempfile.NamedTemporaryFile(suffix='.yaml') as tmp:
        config_path = Path(tmp.name)
        with open(config_path, 'w') as f:
            yaml.dump(db_config, f)
        
        db = DatabaseManager(config_path=config_path)
        yield db 
"""
Setup.py - Smart Trading System v2.0
Configuração para instalação do sistema de trading
"""
from setuptools import setup, find_packages
import os
import sys

# Verifica versão mínima do Python
if sys.version_info < (3, 8):
    sys.exit('Python 3.8+ é necessário para o Smart Trading System')

# Lê README para descrição longa
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Smart Trading System v2.0 - Sistema de Trading Inteligente"

# Lê requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Define dependências principais
CORE_REQUIREMENTS = [
    # Data Science & Analysis
    'pandas>=1.5.0',
    'numpy>=1.21.0',
    'scipy>=1.9.0',
    'scikit-learn>=1.1.0',
    
    # Technical Analysis
    'TA-Lib>=0.4.24',
    'yfinance>=0.1.87',
    
    # Async & Networking
    'aiohttp>=3.8.0',
    'aiofiles>=0.8.0',
    'websockets>=10.0',
    
    # Database
    'sqlalchemy>=1.4.0',
    'alembic>=1.8.0',
    
    # API & Web
    'fastapi>=0.85.0',
    'uvicorn>=0.18.0',
    'streamlit>=1.15.0',
    
    # Visualization
    'matplotlib>=3.5.0',
    'seaborn>=0.11.0',
    'plotly>=5.10.0',
    
    # Utilities
    'python-dotenv>=0.19.0',
    'colorama>=0.4.5',
    'tqdm>=4.64.0',
    'jinja2>=3.1.0',
    
    # Configuration
    'pydantic>=1.10.0',
    'python-multipart>=0.0.5',
    
    # Development & Testing
    'pytest>=7.0.0',
    'pytest-asyncio>=0.20.0',
    'pytest-cov>=4.0.0',
]

OPTIONAL_REQUIREMENTS = {
    'dev': [
        'black>=22.0.0',
        'isort>=5.10.0',
        'flake8>=5.0.0',
        'mypy>=0.991',
        'pre-commit>=2.20.0',
    ],
    'docs': [
        'sphinx>=5.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'myst-parser>=0.18.0',
    ],
    'ml': [
        'tensorflow>=2.10.0',
        'torch>=1.12.0',
        'transformers>=4.21.0',
    ],
    'extra': [
        'redis>=4.3.0',
        'celery>=5.2.0',
        'telegram-bot-api>=0.3.0',
    ]
}

# Combina todos os requirements opcionais para 'all'
OPTIONAL_REQUIREMENTS['all'] = [
    req for reqs in OPTIONAL_REQUIREMENTS.values() 
    for req in reqs if isinstance(req, str)
]

setup(
    # Informações básicas
    name='smart-trading-system',
    version='2.0.0',
    description='Sistema de Trading Inteligente com IA e Confluência',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    
    # Autor e links
    author='Smart Trading Team',
    author_email='admin@smarttrading.com',
    url='https://github.com/smarttrading/smart-trading-system',
    project_urls={
        'Documentation': 'https://docs.smarttrading.com',
        'Source': 'https://github.com/smarttrading/smart-trading-system',
        'Tracker': 'https://github.com/smarttrading/smart-trading-system/issues',
    },
    
    # Classificação
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    
    # Palavras-chave
    keywords=[
        'trading', 'cryptocurrency', 'bitcoin', 'algorithmic-trading',
        'technical-analysis', 'machine-learning', 'finance', 'investment',
        'backtesting', 'portfolio-management', 'risk-management'
    ],
    
    # Licença
    license='MIT',
    
    # Pacotes
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    
    # Arquivos de dados
    package_data={
        'config': ['*.json', '*.yaml', '*.yml'],
        'web': ['templates/*.html', 'static/*'],
        'tests': ['test_data/*'],
    },
    
    # Dependências
    python_requires='>=3.8',
    install_requires=CORE_REQUIREMENTS,
    extras_require=OPTIONAL_REQUIREMENTS,
    
    # Scripts de entrada
    entry_points={
        'console_scripts': [
            'smart-trading=main:main',
            'smart-backtest=main:run_backtest_mode',
            'smart-dashboard=main:run_dashboard_mode',
        ],
    },
    
    # Configuração zip_safe
    zip_safe=False,
    
    # Inclui arquivos do MANIFEST.in
    include_package_data=True,
)


# Post-installation setup
def post_install_setup():
    """Configuração pós-instalação"""
    
    print("\n" + "="*60)
    print("🚀 SMART TRADING SYSTEM v2.0 - Instalação Concluída!")
    print("="*60)
    
    print("\n📋 Próximos passos:")
    print("1. Configure suas credenciais API em config/config.json")
    print("2. Execute: smart-trading --mode dashboard para interface web")
    print("3. Execute: smart-trading --mode backtest para backtesting")
    print("4. Consulte a documentação em: https://docs.smarttrading.com")
    
    print("\n💡 Comandos úteis:")
    print("  smart-trading                    # Modo trading normal")
    print("  smart-trading --mode backtest    # Backtesting")
    print("  smart-trading --mode dashboard   # Dashboard web")
    print("  smart-trading --help             # Ajuda completa")
    
    print("\n🔧 Para desenvolvedores:")
    print("  pip install -e .[dev]           # Instala dependências dev")
    print("  python -m pytest tests/         # Executa testes")
    print("  python main.py --mode backtest  # Teste rápido")
    
    print("\n⚠️  Importante:")
    print("  - Nunca compartilhe suas API keys")
    print("  - Use paper trading antes do real")
    print("  - Comece com capital pequeno")
    print("  - Monitore sempre os resultados")
    
    print("\n" + "="*60)


# Executa setup pós-instalação se chamado diretamente
if __name__ == '__main__':
    # Verifica se foi uma instalação bem-sucedida
    import subprocess
    import sys
    
    try:
        # Tenta importar o módulo principal
        import main
        post_install_setup()
    except ImportError:
        print("⚠️  Instalação pode não ter sido completa. Verifique os requirements.")
    except Exception as e:
        print(f"⚠️  Erro na configuração pós-instalação: {e}")


# Funções auxiliares para desenvolvimento
def create_requirements_file():
    """Cria arquivo requirements.txt atualizado"""
    
    requirements_content = """# Smart Trading System v2.0 - Requirements
# Instalação: pip install -r requirements.txt

# Core Data Science
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0

# Technical Analysis
TA-Lib>=0.4.24
yfinance>=0.1.87

# Async & Networking
aiohttp>=3.8.0
aiofiles>=0.8.0
websockets>=10.0

# Database
sqlalchemy>=1.4.0
alembic>=1.8.0

# API & Web
fastapi>=0.85.0
uvicorn>=0.18.0
streamlit>=1.15.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Utilities
python-dotenv>=0.19.0
colorama>=0.4.5
tqdm>=4.64.0
jinja2>=3.1.0

# Configuration
pydantic>=1.10.0
python-multipart>=0.0.5

# Development & Testing
pytest>=7.0.0
pytest-asyncio>=0.20.0
pytest-cov>=4.0.0

# Optional (uncomment if needed)
# tensorflow>=2.10.0  # For ML features
# torch>=1.12.0       # For PyTorch models
# redis>=4.3.0        # For caching
# celery>=5.2.0       # For task queue
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    
    print("📄 requirements.txt criado com sucesso!")


def create_manifest_file():
    """Cria arquivo MANIFEST.in"""
    
    manifest_content = """# MANIFEST.in - Smart Trading System v2.0
# Inclui arquivos adicionais no pacote

# README e documentação
include README.md
include LICENSE
include CHANGELOG.md
recursive-include docs *

# Configurações
recursive-include config *.json *.yaml *.yml *.ini

# Templates e assets web
recursive-include web/templates *.html
recursive-include web/static *.css *.js *.png *.jpg *.ico

# Dados de teste
recursive-include tests/test_data *

# Scripts
include scripts/*.py
include scripts/*.sh

# Requirements
include requirements*.txt

# Exclui arquivos desnecessários
global-exclude *.pyc
global-exclude *.pyo
global-exclude *~
global-exclude .git*
global-exclude .DS_Store
global-exclude __pycache__
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
"""
    
    with open('MANIFEST.in', 'w', encoding='utf-8') as f:
        f.write(manifest_content)
    
    print("📄 MANIFEST.in criado com sucesso!")


def create_development_files():
    """Cria arquivos úteis para desenvolvimento"""
    
    # pyproject.toml para configuração moderna
    pyproject_content = """[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "smart-trading-system"
version = "2.0.0"
description = "Sistema de Trading Inteligente com IA e Confluência"
authors = [{name = "Smart Trading Team", email = "admin@smarttrading.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"

[tool.black]
line-length = 88
target-version = ['py38']
include = '\\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
"""
    
    with open('pyproject.toml', 'w', encoding='utf-8') as f:
        f.write(pyproject_content)
    
    print("📄 pyproject.toml criado!")


# Executar criação de arquivos se necessário
if len(sys.argv) > 1 and sys.argv[1] == 'create_files':
    create_requirements_file()
    create_manifest_file()
    create_development_files()
    print("\n✅ Todos os arquivos de configuração foram criados!")
"""Project Templates - Framework scaffolding with best practices."""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class FrameworkType(Enum):
    """Supported framework types."""
    REACT = "react"
    VUE = "vue"
    NEXTJS = "nextjs"
    FASTAPI = "fastapi"
    FLASK = "flask"
    DJANGO = "django"
    EXPRESS = "express"
    NESTJS = "nestjs"
    GO_GIN = "go-gin"
    GO_FIBER = "go-fiber"
    RUST_ACTIX = "rust-actix"
    RUST_ROCKET = "rust-rocket"
    SPRING_BOOT = "spring-boot"
    DOTNET = "dotnet"
    VANILLA_PYTHON = "vanilla-python"
    VANILLA_JS = "vanilla-js"


@dataclass
class FileTemplate:
    """Template for a single file."""
    path: str
    content: str
    executable: bool = False


@dataclass
class ProjectTemplate:
    """Complete project template."""
    name: str
    framework: FrameworkType
    description: str
    files: List[FileTemplate]
    dependencies: Dict[str, List[str]]  # package manager -> packages
    setup_commands: List[str]
    dev_commands: Dict[str, str]  # command name -> command


class ProjectTemplateRegistry:
    """Registry of all available project templates."""

    @staticmethod
    def get_template(framework: FrameworkType, project_name: str = "myapp") -> ProjectTemplate:
        """Get template for a framework."""
        templates = {
            FrameworkType.REACT: ProjectTemplateRegistry._react_template,
            FrameworkType.NEXTJS: ProjectTemplateRegistry._nextjs_template,
            FrameworkType.FASTAPI: ProjectTemplateRegistry._fastapi_template,
            FrameworkType.FLASK: ProjectTemplateRegistry._flask_template,
            FrameworkType.DJANGO: ProjectTemplateRegistry._django_template,
            FrameworkType.EXPRESS: ProjectTemplateRegistry._express_template,
            FrameworkType.GO_GIN: ProjectTemplateRegistry._go_gin_template,
            FrameworkType.RUST_ACTIX: ProjectTemplateRegistry._rust_actix_template,
            FrameworkType.VANILLA_PYTHON: ProjectTemplateRegistry._vanilla_python_template,
        }

        template_func = templates.get(framework)
        if not template_func:
            raise ValueError(f"Template not found for framework: {framework}")

        return template_func(project_name)

    @staticmethod
    def list_available_templates() -> List[Dict[str, str]]:
        """List all available templates."""
        return [
            {"framework": "react", "description": "React + TypeScript + Vite"},
            {"framework": "nextjs", "description": "Next.js 14+ with App Router"},
            {"framework": "vue", "description": "Vue 3 + TypeScript + Vite"},
            {"framework": "fastapi", "description": "FastAPI + SQLAlchemy + Pydantic"},
            {"framework": "flask", "description": "Flask + SQLAlchemy + Blueprints"},
            {"framework": "django", "description": "Django + REST Framework"},
            {"framework": "express", "description": "Express + TypeScript + Prisma"},
            {"framework": "nestjs", "description": "NestJS + TypeORM"},
            {"framework": "go-gin", "description": "Go + Gin + GORM"},
            {"framework": "rust-actix", "description": "Rust + Actix-web + Diesel"},
            {"framework": "vanilla-python", "description": "Python with best practices"},
        ]

    # ============================================================
    #  FRONTEND TEMPLATES
    # ============================================================

    @staticmethod
    def _react_template(project_name: str) -> ProjectTemplate:
        """React + TypeScript + Vite template."""
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.REACT,
            description="React application with TypeScript and Vite",
            files=[
                FileTemplate("package.json", f'''{{
  "name": "{project_name}",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {{
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0"
  }},
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.21.0"
  }},
  "devDependencies": {{
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@typescript-eslint/eslint-plugin": "^6.14.0",
    "@typescript-eslint/parser": "^6.14.0",
    "@vitejs/plugin-react": "^4.2.1",
    "eslint": "^8.55.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.5",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }}
}}
'''),
                FileTemplate("tsconfig.json", '''{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
'''),
                FileTemplate("vite.config.ts", '''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
  },
})
'''),
                FileTemplate("index.html", f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
'''),
                FileTemplate("src/main.tsx", '''import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
'''),
                FileTemplate("src/App.tsx", f'''import {{ useState }} from 'react'
import './App.css'

function App() {{
  const [count, setCount] = useState(0)

  return (
    <div className="App">
      <h1>{project_name}</h1>
      <div className="card">
        <button onClick={{() => setCount((count) => count + 1)}}>
          count is {{count}}
        </button>
      </div>
    </div>
  )
}}

export default App
'''),
                FileTemplate("src/index.css", '''body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}
'''),
                FileTemplate("src/App.css", '''.App {
  text-align: center;
  padding: 2rem;
}
'''),
                FileTemplate(".gitignore", '''node_modules
dist
.env
.env.local
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

React + TypeScript + Vite application

## Setup

```bash
npm install
```

## Development

```bash
npm run dev
```

## Build

```bash
npm run build
```
'''),
            ],
            dependencies={
                "npm": ["react", "react-dom", "react-router-dom"]
            },
            setup_commands=["npm install"],
            dev_commands={
                "dev": "npm run dev",
                "build": "npm run build",
                "lint": "npm run lint"
            }
        )

    @staticmethod
    def _nextjs_template(project_name: str) -> ProjectTemplate:
        """Next.js 14+ with App Router template."""
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.NEXTJS,
            description="Next.js application with App Router",
            files=[
                FileTemplate("package.json", f'''{{
  "name": "{project_name}",
  "version": "0.1.0",
  "private": true,
  "scripts": {{
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  }},
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "next": "^14.0.4"
  }},
  "devDependencies": {{
    "@types/node": "^20",
    "@types/react": "^18",
    "@types/react-dom": "^18",
    "typescript": "^5",
    "eslint": "^8",
    "eslint-config-next": "^14.0.4"
  }}
}}
'''),
                FileTemplate("tsconfig.json", '''{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
'''),
                FileTemplate("next.config.js", '''/** @type {import('next').NextConfig} */
const nextConfig = {}

module.exports = nextConfig
'''),
                FileTemplate("src/app/layout.tsx", f'''import type {{ Metadata }} from 'next'
import './globals.css'

export const metadata: Metadata = {{
  title: '{project_name}',
  description: 'Built with Next.js',
}}

export default function RootLayout({{
  children,
}}: {{
  children: React.ReactNode
}}) {{
  return (
    <html lang="en">
      <body>{{children}}</body>
    </html>
  )
}}
'''),
                FileTemplate("src/app/page.tsx", '''export default function Home() {
  return (
    <main>
      <h1>Welcome to Next.js!</h1>
    </main>
  )
}
'''),
                FileTemplate("src/app/globals.css", '''* {
  box-sizing: border-box;
  padding: 0;
  margin: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}
'''),
                FileTemplate(".gitignore", '''node_modules
.next
out
.env*.local
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

Next.js application with App Router

## Setup

```bash
npm install
```

## Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)
'''),
            ],
            dependencies={"npm": ["next", "react", "react-dom"]},
            setup_commands=["npm install"],
            dev_commands={"dev": "npm run dev", "build": "npm run build"}
        )

    # ============================================================
    #  BACKEND TEMPLATES
    # ============================================================

    @staticmethod
    def _fastapi_template(project_name: str) -> ProjectTemplate:
        """FastAPI + SQLAlchemy + Pydantic template."""
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.FASTAPI,
            description="FastAPI application with SQLAlchemy and Pydantic",
            files=[
                FileTemplate("requirements.txt", '''fastapi==0.109.0
uvicorn[standard]==0.25.0
sqlalchemy==2.0.25
pydantic==2.5.3
pydantic-settings==2.1.0
python-dotenv==1.0.0
python-multipart==0.0.6
'''),
                FileTemplate("main.py", f'''"""
{project_name} - FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    openapi_url=f"{{settings.API_V1_STR}}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/health")
async def health_check():
    return {{"status": "healthy"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''),
                FileTemplate("app/__init__.py", ""),
                FileTemplate("app/api/__init__.py", '''from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "API is running"}
'''),
                FileTemplate("app/core/__init__.py", ""),
                FileTemplate("app/core/config.py", '''from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FastAPI App"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./app.db"

    class Config:
        env_file = ".env"

settings = Settings()
'''),
                FileTemplate("app/models/__init__.py", ""),
                FileTemplate("app/schemas/__init__.py", ""),
                FileTemplate("app/crud/__init__.py", ""),
                FileTemplate(".env.example", '''PROJECT_NAME=MyApp
DATABASE_URL=sqlite:///./app.db
'''),
                FileTemplate(".gitignore", '''__pycache__/
*.py[cod]
*$py.class
.env
.venv
env/
venv/
*.db
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

FastAPI application with SQLAlchemy and Pydantic

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

API docs: http://localhost:8000/docs
'''),
            ],
            dependencies={"pip": ["fastapi", "uvicorn", "sqlalchemy", "pydantic"]},
            setup_commands=["pip install -r requirements.txt"],
            dev_commands={"dev": "uvicorn main:app --reload"}
        )

    @staticmethod
    def _flask_template(project_name: str) -> ProjectTemplate:
        """Flask + SQLAlchemy + Blueprints template."""
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.FLASK,
            description="Flask application with SQLAlchemy and Blueprints",
            files=[
                FileTemplate("requirements.txt", '''Flask==3.0.0
Flask-SQLAlchemy==3.1.1
Flask-Migrate==4.0.5
python-dotenv==1.0.0
'''),
                FileTemplate("app.py", f'''"""
{project_name} - Flask Application
"""
from flask import Flask
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''),
                FileTemplate("app/__init__.py", '''from flask import Flask
from app.config import Config
from app.extensions import db

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)

    # Register blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
'''),
                FileTemplate("app/config.py", '''import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
'''),
                FileTemplate("app/extensions.py", '''from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
'''),
                FileTemplate("app/models.py", '''from app.extensions import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'
'''),
                FileTemplate("app/routes.py", '''from flask import Blueprint, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return jsonify({"message": "Flask API is running"})

@main_bp.route('/health')
def health():
    return jsonify({"status": "healthy"})
'''),
                FileTemplate(".env.example", '''SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///app.db
'''),
                FileTemplate(".gitignore", '''__pycache__/
*.py[cod]
.env
venv/
*.db
migrations/
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

Flask application with SQLAlchemy and Blueprints

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

API: http://localhost:5000
'''),
            ],
            dependencies={"pip": ["Flask", "Flask-SQLAlchemy", "Flask-Migrate"]},
            setup_commands=["pip install -r requirements.txt"],
            dev_commands={"dev": "python app.py"}
        )

    @staticmethod
    def _django_template(project_name: str) -> ProjectTemplate:
        """Django + REST Framework template."""
        safe_name = project_name.replace('-', '_')
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.DJANGO,
            description="Django application with REST Framework",
            files=[
                FileTemplate("requirements.txt", '''Django==5.0
djangorestframework==3.14.0
django-cors-headers==4.3.1
python-dotenv==1.0.0
'''),
                FileTemplate("manage.py", f'''#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{safe_name}.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
''', executable=True),
                FileTemplate(f"{safe_name}/__init__.py", ""),
                FileTemplate(f"{safe_name}/settings.py", f'''import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = '{safe_name}.urls'

DATABASES = {{
    'default': {{
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }}
}}

CORS_ALLOW_ALL_ORIGINS = True  # Configure for production
'''),
                FileTemplate(f"{safe_name}/urls.py", '''from django.contrib import admin
from django.urls import path
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "healthy"})

urlpatterns = [
    path('admin/', admin.site.urls),
    path('health/', health_check),
]
'''),
                FileTemplate(".gitignore", '''*.pyc
__pycache__/
db.sqlite3
.env
venv/
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

Django application with REST Framework

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
```

## Run

```bash
python manage.py runserver
```

API: http://localhost:8000
'''),
            ],
            dependencies={"pip": ["Django", "djangorestframework", "django-cors-headers"]},
            setup_commands=["pip install -r requirements.txt", "python manage.py migrate"],
            dev_commands={"dev": "python manage.py runserver"}
        )

    @staticmethod
    def _express_template(project_name: str) -> ProjectTemplate:
        """Express + TypeScript template."""
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.EXPRESS,
            description="Express application with TypeScript",
            files=[
                FileTemplate("package.json", f'''{{
  "name": "{project_name}",
  "version": "1.0.0",
  "main": "dist/index.js",
  "scripts": {{
    "dev": "ts-node-dev --respawn src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js"
  }},
  "dependencies": {{
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1"
  }},
  "devDependencies": {{
    "@types/express": "^4.17.21",
    "@types/cors": "^2.8.17",
    "@types/node": "^20.10.6",
    "typescript": "^5.3.3",
    "ts-node-dev": "^2.0.0"
  }}
}}
'''),
                FileTemplate("tsconfig.json", '''{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
'''),
                FileTemplate("src/index.ts", '''import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'

dotenv.config()

const app = express()
const PORT = process.env.PORT || 3000

app.use(cors())
app.use(express.json())

app.get('/health', (req, res) => {
  res.json({ status: 'healthy' })
})

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})
'''),
                FileTemplate(".env.example", '''PORT=3000
'''),
                FileTemplate(".gitignore", '''node_modules
dist
.env
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

Express + TypeScript application

## Setup

```bash
npm install
```

## Development

```bash
npm run dev
```

## Build & Run

```bash
npm run build
npm start
```
'''),
            ],
            dependencies={"npm": ["express", "cors", "dotenv"]},
            setup_commands=["npm install"],
            dev_commands={"dev": "npm run dev", "build": "npm run build"}
        )

    @staticmethod
    def _go_gin_template(project_name: str) -> ProjectTemplate:
        """Go + Gin framework template."""
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.GO_GIN,
            description="Go application with Gin framework",
            files=[
                FileTemplate("go.mod", f'''module {project_name}

go 1.21

require github.com/gin-gonic/gin v1.9.1
'''),
                FileTemplate("main.go", '''package main

import (
    "github.com/gin-gonic/gin"
    "net/http"
)

func main() {
    r := gin.Default()

    r.GET("/health", func(c *gin.Context) {
        c.JSON(http.StatusOK, gin.H{
            "status": "healthy",
        })
    })

    r.Run(":8080")
}
'''),
                FileTemplate(".gitignore", '''*.exe
*.exe~
*.dll
*.so
*.dylib
*.test
*.out
vendor/
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

Go application with Gin framework

## Setup

```bash
go mod download
```

## Run

```bash
go run main.go
```

API: http://localhost:8080
'''),
            ],
            dependencies={"go": ["github.com/gin-gonic/gin"]},
            setup_commands=["go mod download"],
            dev_commands={"dev": "go run main.go", "build": "go build"}
        )

    @staticmethod
    def _rust_actix_template(project_name: str) -> ProjectTemplate:
        """Rust + Actix-web template."""
        safe_name = project_name.replace('-', '_')
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.RUST_ACTIX,
            description="Rust application with Actix-web",
            files=[
                FileTemplate("Cargo.toml", f'''[package]
name = "{project_name}"
version = "0.1.0"
edition = "2021"

[dependencies]
actix-web = "4.4"
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
'''),
                FileTemplate("src/main.rs", '''use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::Serialize;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

async fn health() -> impl Responder {
    HttpResponse::Ok().json(HealthResponse {
        status: "healthy".to_string(),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Server running on http://localhost:8080");

    HttpServer::new(|| {
        App::new()
            .route("/health", web::get().to(health))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
'''),
                FileTemplate(".gitignore", '''target/
Cargo.lock
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

Rust application with Actix-web

## Setup & Run

```bash
cargo run
```

## Build

```bash
cargo build --release
```

API: http://localhost:8080
'''),
            ],
            dependencies={"cargo": ["actix-web", "serde"]},
            setup_commands=[],
            dev_commands={"dev": "cargo run", "build": "cargo build --release"}
        )

    @staticmethod
    def _vanilla_python_template(project_name: str) -> ProjectTemplate:
        """Vanilla Python with best practices."""
        safe_name = project_name.replace('-', '_')
        return ProjectTemplate(
            name=project_name,
            framework=FrameworkType.VANILLA_PYTHON,
            description="Python project with best practices",
            files=[
                FileTemplate("requirements.txt", '''pytest==7.4.3
black==23.12.1
mypy==1.7.1
'''),
                FileTemplate(f"{safe_name}/__init__.py", f'''"""
{project_name}
"""

__version__ = "0.1.0"
'''),
                FileTemplate(f"{safe_name}/main.py", '''"""Main module."""

def main():
    print("Hello from main!")

if __name__ == "__main__":
    main()
'''),
                FileTemplate("tests/__init__.py", ""),
                FileTemplate("tests/test_main.py", f'''from {safe_name}.main import main

def test_main():
    # Add your tests here
    assert True
'''),
                FileTemplate("setup.py", f'''from setuptools import setup, find_packages

setup(
    name="{project_name}",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
)
'''),
                FileTemplate(".gitignore", '''__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.pytest_cache/
.DS_Store
'''),
                FileTemplate("README.md", f'''# {project_name}

Python project with best practices

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python -m {safe_name}.main
```

## Test

```bash
pytest
```
'''),
            ],
            dependencies={"pip": ["pytest", "black", "mypy"]},
            setup_commands=["pip install -r requirements.txt", "pip install -e ."],
            dev_commands={"test": "pytest", "format": "black .", "typecheck": "mypy ."}
        )


def generate_project(
    framework: str,
    output_dir: str,
    project_name: str = "myapp"
) -> Dict[str, str]:
    """
    Generate a project from template.

    Args:
        framework: Framework name (e.g., "react", "fastapi")
        output_dir: Directory to generate project in
        project_name: Name of the project

    Returns:
        Dictionary mapping file paths to content
    """
    import os

    # Get framework enum
    framework_map = {ft.value: ft for ft in FrameworkType}
    framework_type = framework_map.get(framework)

    if not framework_type:
        raise ValueError(f"Unknown framework: {framework}")

    # Get template
    template = ProjectTemplateRegistry.get_template(framework_type, project_name)

    # Generate files
    files = {}
    for file_template in template.files:
        full_path = os.path.join(output_dir, file_template.path)
        files[full_path] = file_template.content

    return files

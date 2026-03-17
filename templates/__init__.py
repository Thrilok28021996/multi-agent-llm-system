"""Project Templates Package"""

from .project_templates import (
    FrameworkType,
    ProjectTemplate,
    ProjectTemplateRegistry,
    generate_project
)

__all__ = [
    'FrameworkType',
    'ProjectTemplate',
    'ProjectTemplateRegistry',
    'generate_project'
]

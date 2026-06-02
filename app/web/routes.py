"""Web interface routes for MXMAP-X."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main prediction interface."""
    return templates.TemplateResponse(request, "index.html")


@router.get("/optimize", response_class=HTMLResponse)
async def optimize(request: Request):
    """Multi-objective optimization interface."""
    return templates.TemplateResponse(request, "optimize.html")


@router.get("/explore", response_class=HTMLResponse)
async def explore(request: Request):
    """Chemistry space exploration interface."""
    return templates.TemplateResponse(request, "explore.html")


@router.get("/recipes", response_class=HTMLResponse)
async def recipes(request: Request):
    """Recipe card interface."""
    return templates.TemplateResponse(request, "recipe.html")


@router.get("/electrochromic", response_class=HTMLResponse)
async def electrochromic(request: Request):
    """Electrochromic visualization interface."""
    return templates.TemplateResponse(request, "electrochromic.html")


@router.get("/filtering", response_class=HTMLResponse)
async def filtering(request: Request):
    """AC-line filtering interface."""
    return templates.TemplateResponse(request, "filtering.html")


@router.get("/printing", response_class=HTMLResponse)
async def printing(request: Request):
    """Printing/process-aware design interface."""
    return templates.TemplateResponse(request, "printing.html")

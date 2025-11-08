"""Web interface routes for MXMAP-X."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main prediction interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/optimize", response_class=HTMLResponse)
async def optimize(request: Request):
    """Multi-objective optimization interface."""
    return templates.TemplateResponse("optimize.html", {"request": request})


@router.get("/explore", response_class=HTMLResponse)
async def explore(request: Request):
    """Chemistry space exploration interface."""
    return templates.TemplateResponse("explore.html", {"request": request})


@router.get("/recipes", response_class=HTMLResponse)
async def recipes(request: Request):
    """Recipe card interface."""
    return templates.TemplateResponse("recipe.html", {"request": request})

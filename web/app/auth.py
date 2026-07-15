from datetime import datetime, timedelta, timezone
from typing import Any

from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import User


# Authlib OAuth registry.
oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

security_scheme = HTTPBearer(auto_error=False)


class OAuthProfileError(ValueError):
    """Raised when Google returns an ID token without required profile claims."""


def normalize_google_profile(user_info: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the Google OpenID Connect claims used by this app."""

    google_id = user_info.get("sub")
    email = user_info.get("email")
    name = user_info.get("name")

    if not google_id:
        raise OAuthProfileError("Google ID token is missing the subject claim.")
    if not email:
        raise OAuthProfileError("Google ID token is missing an email address.")
    if not name:
        raise OAuthProfileError("Google ID token is missing a display name.")

    return {
        "google_id": str(google_id),
        "email": str(email).lower(),
        "name": str(name),
        "picture": user_info.get("picture"),
        "email_verified": bool(user_info.get("email_verified", False)),
    }


def create_access_token(user: User) -> str:
    """Create a short-lived JWT containing the user identity claims."""

    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    payload = {
        "user_id": user.id,
        "email": user.email,
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_access_token(token: str) -> dict[str, Any]:
    """Decode and validate an access token, raising a clear error when invalid."""

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except ExpiredSignatureError as exc:
        raise ValueError("Token has expired.") from exc
    except JWTError as exc:
        raise ValueError("Invalid access token.") from exc

    if not payload.get("user_id") or not payload.get("email"):
        raise ValueError("Token payload is invalid.")

    return payload


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> User:
    """Resolve the currently authenticated user from a bearer token or cookie."""

    token = None
    if credentials is not None:
        token = credentials.credentials
    else:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )

    try:
        payload = verify_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is invalid.",
        )

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Authenticated user no longer exists.",
        )

    return user


async def upsert_google_user(
    db: AsyncSession,
    profile: dict[str, Any],
) -> User:
    """Create a new Google user or refresh an existing user's login metadata."""

    now = datetime.utcnow()

    try:
        result = await db.execute(
            select(User).where(User.google_id == profile["google_id"])
        )
        user = result.scalar_one_or_none()

        if user is None:
            user = User(
                google_id=profile["google_id"],
                email=profile["email"],
                name=profile["name"],
                picture=profile["picture"],
                email_verified=profile["email_verified"],
                last_login=now,
            )
            db.add(user)
        else:
            user.email = profile["email"]
            user.name = profile["name"]
            user.picture = profile["picture"]
            user.email_verified = profile["email_verified"]
            user.last_login = now

        await db.commit()
        await db.refresh(user)
        return user

    except IntegrityError:
        await db.rollback()
        raise
    except SQLAlchemyError:
        await db.rollback()
        raise

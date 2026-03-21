import time
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, Request

import jwt

from .settings import settings

ALGORITHM = "HS256"


@dataclass
class TokenPayload:
    user_id: str
    role: str = "user"
    exp: float = 0.0


def create_token(user_id: str, role: str = "user") -> str:
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": time.time() + settings.jwt_expire_seconds,
        "iat": time.time(),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    try:
        data = jwt.decode(token, settings.jwt_secret_key, algorithms=[ALGORITHM])
        return TokenPayload(
            user_id=data["user_id"],
            role=data.get("role", "user"),
            exp=data["exp"],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=str(e))


def get_user_from_request(request: Request) -> Optional[TokenPayload]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return decode_token(auth.split(" ", 1)[1])


def require_auth(request: Request) -> TokenPayload:
    if not settings.auth_enabled:
        return TokenPayload(user_id="anonymous")
    user = get_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

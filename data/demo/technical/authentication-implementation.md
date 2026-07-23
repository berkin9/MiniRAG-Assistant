# Atlas Authentication Implementation

This is fictional demo data created for the MiniRAG Assistant portfolio project.

Atlas uses OpenID Connect (OIDC) authentication with a fictional corporate
identity provider. After successful authentication, the backend issues an access
token that expires after 15 minutes.

Refresh tokens are stored in secure HTTP-only cookies. They are never made
available to browser JavaScript. The backend rotates refresh tokens when they
are used and validates the OIDC issuer, audience, signature, and expiration.

Authorization uses three application roles:

- `viewer` can read their own requests.
- `support_agent` can work on assigned customer requests.
- `administrator` can manage application configuration and role assignments.

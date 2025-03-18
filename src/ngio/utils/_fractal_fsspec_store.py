import fsspec.implementations.http


def fractal_fsspec_store(
    url: str, fractal_token: str | None = None, client_kwargs: dict | None = None
) -> fsspec.mapping.FSMap:
    """Simple function to get an http fsspec store from a url."""
    client_kwargs = {} if client_kwargs is None else client_kwargs
    if fractal_token is not None:
        client_kwargs["headers"] = {"Authorization": f"Bearer {fractal_token}"}
    fs = fsspec.implementations.http.HTTPFileSystem(client_kwargs=client_kwargs)
    store = fs.get_mapper(url)
    return store

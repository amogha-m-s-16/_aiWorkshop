

Streamlit apps deploy easily to [Streamlit Community Cloud](https://streamlit.io/cloud)
or any host that can run a long-lived Python process (a small VM, Render,
Railway, etc.) — just make sure `ANTHROPIC_API_KEY` is set as an
environment variable there, and that `data/` is on persistent storage if
you want recordings and trained models to survive restarts.

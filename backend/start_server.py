import sys, traceback
try:
    import uvicorn
    print('Starting server...')
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, log_level='debug')
except SystemExit as e:
    print(f'SystemExit: {e}')
except Exception as e:
    traceback.print_exc()
except BaseException as e:
    print(f'BaseException: {type(e).__name__}: {e}')

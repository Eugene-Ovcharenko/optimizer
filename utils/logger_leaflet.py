
log_leaflet = None

def configure_log_leaflet(baseName):
    global log_leaflet
    logger = open(baseName + ".log", "w")
    logger.close()
    log_leaflet = open(baseName + ".log", "a")

def log_message(message):
    global log_leaflet
    if log_leaflet:
        log_leaflet.write(message + '\n')
        log_leaflet.flush()

def cleanup_log_leaflet():
    global log_leaflet
    log_leaflet = None

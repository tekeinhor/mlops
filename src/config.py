from dynaconf import Dynaconf

settings = Dynaconf(envvar_prefix="APP", settings_files=['conf/config.toml'],)

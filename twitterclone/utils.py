from uuid import UUID

def validate_uuid4(uuid_string):
	try:
		val = UUID(uuid_string, version=4)
	except ValueError:
		return False
	return val.hex == uuid_string
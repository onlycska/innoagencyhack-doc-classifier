-- added user and return idusers
INSERT INTO postgres.public.users (username, userpass)
VALUES
('name_user', 'pass_user')
RETURNING idusers;
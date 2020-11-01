-- add user and return idusers
INSERT INTO postgres.public.users (username, userpass)
VALUES
('name_user', 'pass_user')
RETURNING idusers;

--add image
INSERT INTO postgres.public.image (url, recordid)
VALUES
('url', 'id');

--get user
SELECT postgres.public.users.username 
FROM users
WHERE username LIKE '%username%';

--search by state
SELECT postgres.public.extracteddata.status 
FROM extracteddata
WHERE status LIKE '%status%';
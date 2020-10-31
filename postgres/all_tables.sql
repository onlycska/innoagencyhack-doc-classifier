CREATE SCHEMA public;

CREATE TABLE classifiers (
    classifierId bigserial NOT NULL UNIQUE PRIMARY KEY,
    autoPublish boolean,
    --publishedModelId bigint,
    created timestamp,
    minProbabilility real,
    name text,
    status int,
    published timestamp,
    useFixedForms boolean
);

CREATE TABLE metricses (
    metricsId bigserial NOT NULL UNIQUE PRIMARY KEY,
    accuracy double precision,
    f1Measure double precision,
    precission double precision,
    recall double precision,
    testDuration double precision,
    trainCount int,
    trainDuration double precision,
    testCount int,
    eter double precision
);

CREATE TABLE classifierModels (
    classifierModelId bigserial NOT NULL  UNIQUE PRIMARY KEY,
    classifierId bigint, --key
    created timestamp,
    --metricsId bigint,
    --modelDataId bigint,
    trainErrors text,
    recommendedThreshold double precision,
    FOREIGN KEY (classifierId) REFERENCES classifiers (classifierId) ON DELETE CASCADE
);

CREATE TABLE trainTasks (
    taskId bigserial NOT NULL UNIQUE PRIMARY KEY,
    classifierId bigint, --key
    classifierModelId bigint, --key
    finished timestamp,
    result text,
    started timestamp,
    state int,
    FOREIGN KEY (classifierId) REFERENCES classifiers (classifierId) ON DELETE CASCADE,
    FOREIGN KEY (classifierModelId) REFERENCES classifierModels (classifierModelId) ON DELETE CASCADE
);

CREATE TABLE classes (
    classId bigserial NOT NULL UNIQUE PRIMARY KEY,
    classifierModelId bigint, --key
    displayName text,
    externalId text,
    metricsId bigint, --key
    name text,
    isFixedForms boolean,
    FOREIGN KEY (classifierModelId) REFERENCES classifierModels (classifierModelId) ON DELETE CASCADE,
    FOREIGN KEY (metricsId) REFERENCES metricses (metricsId) ON DELETE CASCADE
);

CREATE TABLE classificationResults (
    classificationResultId bigserial NOT NULL UNIQUE PRIMARY KEY,
    classifield timestamp,
    classifierId bigint, --key
    classifierModelId bigint, --key
    duration real,
    error text,
    --expectedClassId bigint,
    --predictedClassId bigint,
    predictedProbablility real,
    FOREIGN KEY (classifierId) REFERENCES classifiers (classifierId) ON DELETE CASCADE,
    FOREIGN KEY (classifierModelId) REFERENCES classifierModels (classifierModelId) ON DELETE CASCADE
);

CREATE TABLE classResults (
    classResultId bigserial NOT NULL UNIQUE PRIMARY KEY,
    classId bigint, --key
    classificationResultId bigint, --key
    probability real,
    FOREIGN KEY (classId) REFERENCES classes (classId) ON DELETE CASCADE,
    FOREIGN KEY (classificationResultId) REFERENCES classificationResults (classificationResultId) ON DELETE CASCADE
);

--логины пароли юзеров
CREATE TABLE users (
    idUsers bigserial NOT NULL UNIQUE PRIMARY KEY,
    userName text,
    userPass text
);

--токены 
CREATE TABLE refreshtokens (
    idUsers bigint,
    valuetoken text,
    FOREIGN KEY (idUsers) REFERENCES users (idUsers) ON DELETE CASCADE
);

--текст со сканов
CREATE TABLE record (
    recordId bigserial NOT NULL UNIQUE PRIMARY KEY,
    textValue text,
    categoryId bigint
);

--классификации документов
CREATE TABLE category (
    categoryId bigserial NOT NULL UNIQUE PRIMARY KEY,
    name text
);

--хранение картинок
CREATE TABLE image(
    url text,
    recordId bigint,
    FOREIGN KEY (recordId) REFERENCES record(recordId)
);
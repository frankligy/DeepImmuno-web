from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField,BooleanField
from flask_wtf.file import FileField


class QueryForm(FlaskForm):
    peptide = StringField('peptide')
    mhc = StringField('MHC')
    checkbox = BooleanField()
    file_upload = FileField()
    submit_button = SubmitField('Query')





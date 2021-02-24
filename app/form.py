from flask_wtf import FlaskForm
from wtforms import TextAreaField,SubmitField


class QueryForm(FlaskForm):
    query = TextAreaField('Please input your queried peptide and MHC molecule')
    submit = SubmitField('submit')




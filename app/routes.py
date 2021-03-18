from flask import render_template,flash,redirect,url_for,request,send_from_directory,send_file,session
from app import app
from app.program import computing_s,computing_m,hla_convert,svg_path,check_peptide,check_mhc,wrapper_file_process, binding_score_from_mhcflurry_s,check_upload
from app.form import QueryForm
import sys
import os


@app.route('/result')
def result():
    peptide = request.args.get('peptide')
    mhc = request.args.get('mhc')
    is_checked = request.args.get('is_checked')  # str
    score,flag= computing_s(peptide,mhc)
    if is_checked == 'True':
        binding = binding_score_from_mhcflurry_s(peptide,mhc)
    else:
        binding = "NA"
    p,m,i,b = computing_m(peptide,mhc,is_checked)
    m3 = hla_convert(mhc)
    path = ["/static/{0}_positive_9.png".format(m3),"/static/{0}_negative_9.png".format(m3),"/static/{0}_positive_10.png".format(m3),"/static/{0}_negative_9.png".format(m3)]
    return render_template('result.html',peptide=peptide,mhc=mhc,score=score,p=p,m=m,i=i,m3=m3,path=path,binding=binding,b=b,flag=flag)


@app.route('/download')
def download():
    return send_file("download/result.txt",as_attachment=True)


@app.route('/learn_more')
def learn_more():
    return render_template('learn_more.html')




@app.route('/',methods=['GET','POST'])
def home():
    form = QueryForm()
    if request.method=='POST':
        session['peptide'] = form.peptide.data
        session['mhc'] = form.mhc.data
        session['checkbox'] = form.checkbox.data  # boolean
        if check_peptide(session.get('peptide')) and check_mhc(session.get('mhc')) and form.file_upload.data is None:
            '''
            this condition means the users have valid peptide and mhc input, and no file uploaded
            '''
            return redirect(url_for('result',peptide=session.get('peptide'),mhc=session.get('mhc'),is_checked=session.get('checkbox')))

        elif (form.file_upload.data is None and check_peptide(session.get('peptide'))==False) or (form.file_upload.data is None and check_mhc(session.get('mhc'))==False):
            '''
            this condition means the users have either invalid peptide or mhc input, and no file uploaded
            '''
            flash("Please check your peptide and MHC input!")
            flash("Peptide: length of 9 or 10, valid amino acid one letter!")
            flash("MHC: format like HLA-A*0201")
            return redirect(url_for('home'))


        else:
            '''
            this condition means the users have file uploaded
            '''
            if check_peptide(session.get('peptide'))==False and check_mhc(session.get('mhc'))==False:
                '''
                no peptide and mhc input
                '''
                uploaded_file = form.file_upload.data  # either a filestorage object or NoneType
                if sys.getsizeof(uploaded_file) < 500:
                    uploaded_file.save("./uploaded/multiple_query.txt")
                    is_checked = session.get('checkbox')
                    is_right = check_upload('./uploaded/multiple_query.txt')
                    print(is_right)
                    cond = wrapper_file_process(str(is_checked))
                    if not cond or not is_right:
                        flash("please check your file format:")
                        flash("file should contains two columns,delimiter is comma")
                        flash("first column is the peptide, make sure length is 9 or 10, and only contains valid amino acid letter")
                        flash("second column is the HLA, make sure the format like HLA-A*0201 ")
                        return redirect(url_for('home'))
                    else:
                        return redirect(url_for('download'))
                else:  # the file > certain size limit
                    flash("Currently we only support file less than 5MB")
                    return redirect(url_for('home'))
            else:
                '''
                have peptide or mhc input
                '''
                flash("You have uploaded files and inputted peptide or mhc")
                flash("please do either single query or multiple query")
                return redirect(url_for('home'))
    else:
        return render_template('submit.html',form=form)   











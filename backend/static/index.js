function set_checkpoint(value){
    let checkpoint_val = $("#checkpoint_val");
    checkpoint_val.val(value);
    checkpoint_val.text(value);
    $("#submit").removeAttr("disabled");
}

function submitdata() {
    let res_element = $("#results");
    res_element.empty();
    res_element.attr("hidden");
    $.getJSON( "_predict", {
    seq_str: btoa($("#input_seq").val()),
    checkpoint_val: $("#checkpoint_val").val(),
    format: "json"
  })
    .done(function( data ) {
     let results = data.data.results;
        let res_element = $("#results");
        res_element.removeAttr("hidden");
        res_element.empty();
        results.forEach(function (element) {
            console.log(element);
            res_element.append(
                "<button type=\"button\" class=\"list-group-item list-group-item-action\">" + element + "</button>"
            );
        });
      });
}